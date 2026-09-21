public class Phone {
	private final String unformattedNumber;

	public String getNumber() {
		return unformattedNumber.substring(6,10);
	}
}
